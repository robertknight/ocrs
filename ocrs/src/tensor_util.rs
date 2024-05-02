use std::borrow::Cow;

use rten_tensor::prelude::*;
use rten_tensor::{MutLayout, TensorBase};

/// Convert an owned tensor or view into one which uses a [Cow] for storage.
///
/// This is useful for code that wants to conditionally copy a tensor, as this
/// trait can be used to convert either an owned copy or view to the same type.
pub trait IntoCow {
    type Cow;

    fn into_cow(self) -> Self::Cow;
}

impl<'a, T, L: MutLayout> IntoCow for TensorBase<T, &'a [T], L>
where
    [T]: ToOwned,
{
    type Cow = TensorBase<T, Cow<'a, [T]>, L>;

    fn into_cow(self) -> Self::Cow {
        TensorBase::from_data(self.shape(), Cow::Borrowed(self.non_contiguous_data()))
    }
}

impl<T: Clone + 'static, L: MutLayout> IntoCow for TensorBase<T, Vec<T>, L>
where
    [T]: ToOwned<Owned = Vec<T>>,
{
    type Cow = TensorBase<T, Cow<'static, [T]>, L>;

    fn into_cow(self) -> Self::Cow {
        let layout = self.layout().clone();
        TensorBase::from_data(layout.shape(), Cow::Owned(self.into_data()))
    }
}
